#!/usr/bin/perl

# * NiuTrans.NMT v1.0                   * #
# * Author: Qiang Li,                   * #
# * Email : liqiangneu@gmail.com        * #
# * Date  : 10/30/2016                  * #
# * Time  : 15:31                       * #
# * NiuTrans.NMT-postprocess-unk.pl     * #

use strict;


my $logo = "Postprocess cwmt2017\n";
print STDERR $logo;

my %param;
my %dict;

GetParameter(@ARGV);

LoadDict();
PostProcessUnk();




#################################################
sub PostProcessUnk
{
  print STDERR "Start replacing cwmt2017 ...\n";
  open (INPUT, "<", $param{"-input"}) or die "Error: can not read $param{'-input'}.\n";
  open (OUTPUT, ">", $param{"-output"}) or die "Error: can not write $param{'-output'}.\n";

  my $line_num = 0;
  while (my $line = <INPUT>)
  {
    ++$line_num;

    $line =~ s/[\r\n]//g;
    $line =~ s/\([\sA-Za-z]+\)//g;
    $line =~ s/\s+/ /g;

    my @words = split /\s+/, $line;
    my $new_line = "";
    foreach my $word (@words)
    {
      if (exists $dict{$word})
      {
        $new_line .= " ".$dict{$word};
      } 
      else
      {
        $new_line .= " ".$word;
      }
    }
    $new_line =~ s/^\s+//g;
    $new_line =~ s/\s+$//g;
    $new_line =~ s/\s+/ /g;
    print OUTPUT $new_line."\n";

    if ($line_num % 1000 == 0)
    {
      print STDERR "\r$line_num";
    }
  }  
  print STDERR "\r$line_num\n";
  
  close (INPUT);
  close (OUTPUT);
}


#################################################
sub LoadDict()
{
  print STDERR "Start loading dict ...\n";
  open (INPUTDICT, "<", $param{"-dict"}) or die "Error: can not read $param{'-dict'}.\n";
  my $line_num = 0;
  while (my $line = <INPUTDICT>)
  {
    ++$line_num;
    $line =~ s/[\r\n]//g;
    my @fields = split /\t/, $line;
    if (scalar(@fields) ne 2)
    {
      print STDERR "\nError: format error in $line_num\n";
      next;
    }
    
    $dict{$fields[0]} = $fields[1];
    
    if ($line_num % 100 == 0)
    {
      print STDERR "\r  $line_num";
    }
  }
  print STDERR "\r  $line_num\n";
  my $dict_size = keys %dict;
  print STDERR "  dict_size: ".$dict_size."\n";
  close (INPUTDICT);
}


#################################################
sub GetParameter
{
  if( ( scalar( @_ ) < 2 ) || ( scalar( @_ ) % 2 != 0 ) )
  {
    print STDERR "[USAGE]\n".
                 "        LiNMT-postprocess-cwmt2017.pl   [OPTIONS]\n".
                 "[OPTION]\n".
                 "            -dict  : parallel dict\n".
                 "           -input  : input file\n".
                 "          -output  : output file\n".
                 "[EXAMPLE]\n".
                 "        perl LiNMT-postprocess-cwmt2017.pl\n";
    exit( 0 );
  }
          
  my $pos;
  for( $pos = 0; $pos < scalar( @_ ); ++$pos )
  {
    my $key = $ARGV[ $pos ];
    ++$pos;
    my $value = $ARGV[ $pos ];
    $param{ $key } = $value;
  }

  if (!exists $param{"-dict"} || !exists $param{"-input"} || !exists $param{"-output"})
  {
    print STDERR "ERROR: '-dict', '-input', or '-output' must be assigned!\n";
    exit(1);
  }
}


#################################################
sub Ssystem
{
  print STDERR "Running: @_\n";
  system( @_ );
  if( $? == -1 )
  {
    print STDERR "Error: Failed to execute: @_\n  $!\n";
    exit( 1 );
  }
  elsif( $? & 127 )
  {
    printf STDERR "Error: Execution of: @_\n   die with signal %d, %s coredump\n",
                  ($? & 127 ), ( $? & 128 ) ? 'with' : 'without';
    exit( 1 );
  }
  else
  {
    my $exitcode = $? >> 8;
    print STDERR "Exit code: $exitcode\n" if $exitcode;
    return ! $exitcode;
  }         
}

