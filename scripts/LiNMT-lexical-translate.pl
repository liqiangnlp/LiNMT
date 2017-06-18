#!/usr/bin/perl

# * NiuTrans.NMT v1.0                   * #
# * Author: Qiang Li,                   * #
# * Email : liqiangneu@gmail.com        * #
# * Date  : 10/30/2016                  * #
# * Time  : 15:31                       * #
# * NiuTrans.NMT-preprocess-lex.pl      * #

use strict;


my $logo = "Lexical Translate\n";
print STDERR $logo;

my %param;

GetParameter(@ARGV);
LexicalTranslate();



sub LexicalTranslate 
{
  print STDERR "Load lexical dictionary...\n";
  my %lexical_dictionary;
  open (DICT, "<", $param{"-dict"}) or die "Error: can not read $param{'-dict'}!\n";
  my $line_num = 0;
  while (my $line = <DICT>)
  {
    ++$line_num;

    $line =~ s/[\r\n]//g;
    my @fields = split /\s+/, $line;
    if (scalar(@fields) ne 3)
    {
      print STDERR "Warning: format error\n";
      next;
    }

    $lexical_dictionary{ $fields[0] } = $fields[1];
    if ($line_num % 1000 == 0)
    {
      print STDERR "\r  Processed $line_num lines";
    }
  }
  close (DICT);
  print STDERR "\r  Processed $line_num lines\n";
  my $dict_size = keys %lexical_dictionary;
  print STDERR "  DICT-SIZE=$dict_size\n\n";

  open (INPUT, "<", $param{"-input"}) or die "Error: can not read $param{'-input'}\n";
  open (OUTPUT, ">", $param{"-output"}) or die "Error: can not write $param{'-output'}\n";
  $line_num = 0;
  while (my $line = <INPUT>)
  {
    ++$line_num;
    $line =~ s/[\r\n]//g;
    my @words = split /\s+/, $line;
    my $output_line = "";
    foreach my $word (@words)
    {
      if (exists $lexical_dictionary{$word})
      {
        if ($lexical_dictionary{$word} eq 'NULL')
        {
          $output_line .= " #";
        }
        else 
        {
          $output_line .= " ".$lexical_dictionary{$word};
        }
      }
      else
      {
        $output_line .= " #";
      }
    }
    $output_line =~ s/^\s+//g;
    $output_line =~ s/\s+$//g;
    print OUTPUT $output_line."\n";

    if ($line_num % 1000 == 0)
    {
      print STDERR "\r  Processed $line_num lines";
    }
  }
  print STDERR "\r  Processed $line_num lines\n";
  
  close (INPUT);
  close (OUTPUT);
}



sub GetParameter
{
  if( ( scalar( @_ ) < 2 ) || ( scalar( @_ ) % 2 != 0 ) )
  {
    print STDERR "[USAGE]\n".
                 "        LiNMT-lexical-translate.pl   [OPTIONS]\n".
                 "[OPTION]\n".
                 "             -dict : Lexical dictionary\n".
                 "            -input : Input lexcial table\n".
                 "           -output : Preprocessed lexical table\n".
                 "[EXAMPLE]\n".
                 "        perl LiNMT-lexical-translate.pl\n";
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

  if (!exists $param{"-input"} || !exists $param{"-output"})
  {
    print STDERR "ERROR: '-dict', '-input', or '-output' must be assigned!\n";
    exit(1);
  }
}



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



