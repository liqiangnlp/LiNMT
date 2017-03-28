#!/usr/bin/perl

# * LiNMT v1.0                          * #
# * Author: Qiang Li,                   * #
# * Email : liqiangneu@gmail.com        * #
# * Date  : 10/30/2016                  * #
# * Time  : 15:31                       * #
# * training recurrent neural network   * #

use strict;


my $logo = "Training rnn model\n";
print STDERR $logo;

my %param;
my %option;

GetParameter(@ARGV);
ReadConfigFile($param{ "-config" });

my $exec = "../bin/LiNMT";
my $opt = "";

my $key;
my $value;
while (($key, $value) = each %option) {
  $opt .= " $key $value";
}

my $command = $exec.$opt;
Ssystem($command);



sub GetParameter
{
  if( ( scalar( @_ ) < 2 ) || ( scalar( @_ ) % 2 != 0 ) )
  {
    print STDERR "[USAGE]\n".
                 "        LiNMT.pl              [OPTIONS]\n".
                 "[OPTION]\n".
                 "    -config   :  Configuration file\n".
                 "[EXAMPLE]\n".
                 "   perl LiNMT-nmt-training.pl\n";
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

  if( !exists $param{ "-config" } )
  {
    print STDERR "ERROR: '-config' must be used!\n";
    exit(1);
  }
}



sub ReadConfigFile
{
  print STDERR "Error: Config file does not exist!\n" if( scalar( @_ ) != 1 );
  $_[0] =~ s/\\/\//g;
  open( CONFIGFILE, "<".$_[0] ) or die "\nError: Can not read file $_[0] \n";
  print STDERR "Read $param{ \"-config\" } ";
  my $configFlag = 0;
  my $appFlag = 0;
  my $lineNo = 0;
  while( <CONFIGFILE> )
  {
    s/[\r\n]//g;
    next if (/^#/);
    next if /^( |\t)*$/;
    if( /param(?: |\t)*=(?: |\t)*"([\w\-]*)"(?: |\t)*value="([\w\/\-. :]*)"(?: |\t)*/ )
    {
      ++$lineNo;
      $option{$1} = $2;
      print STDERR ".";
    }
  }
  close( CONFIGFILE ); 
  print STDERR " Over.\n\n";
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

